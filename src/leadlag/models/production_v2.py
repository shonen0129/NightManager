"""Production v2 portfolio construction module.

Public API: ProductionV2Model, load_pit_ir_history,
ProductionV2Model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.pit_lake import (
    MarketSnapshot,
    PITDataLake,
    validate_production_decision_inputs,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2 import (
    VERSION,
    _build_current_prices_from_df_exec,
    generate_v2_production_portfolio_from_distribution,
    load_pit_ir_history,
)
from leadlag.models.v2 import (
    _compute_ondemand as _v2_compute_ondemand,
)
from leadlag.models.v2 import (
    _decide as _v2_decide,
)
from leadlag.models.v2 import (
    _file_cache_or_flat as _v2_file_cache_or_flat,
)
from leadlag.models.v2 import (
    _multi_horizon_scores as _v2_multi_horizon_scores,
)
from leadlag.models.v2 import (
    _resolve_current_index as _v2_resolve_current_index,
)
from leadlag.models.v2 import (
    compute_distribution as _v2_compute_distribution,
)
from leadlag.models.v2.overlay_applier import _apply_overlay as _v2_apply_overlay
from leadlag.utils.cache_manager import CacheManager
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_GROSS",
    "COST_BPS_PER_GROSS",
    "LONG_COUNT",
    "SHORT_COUNT",
    "ProductionV2Model",
    "VERSION",
    "generate_v2_production_portfolio_from_distribution",
    "load_pit_ir_history",
    "_build_current_prices_from_df_exec",
]

# Default constants (mirror ProductionV2RunConfig Pydantic defaults).
BASELINE_GROSS = 2.0
COST_BPS_PER_GROSS = 10.0
LONG_COUNT = 5
SHORT_COUNT = 5


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

    def _file_cache_or_flat(
        self,
        trade_date: str,
        gap_input_dir: Path | None,
    ) -> PortfolioDecision:
        return _v2_file_cache_or_flat(self, trade_date, gap_input_dir)

    def decide(
        self,
        trade_date: str | None = None,
        gap_input_dir: str | Path | None = None,
        df_exec: pd.DataFrame | None = None,
        current_prices: dict[str, float] | None = None,
        overlay_enabled: bool = True,
        use_file_cache: bool = True,
        lake: PITDataLake | None = None,
        snapshot: MarketSnapshot | None = None,
        inputs: DecisionInputs | None = None,
    ) -> PortfolioDecision:
        if inputs is not None:
            if gap_input_dir is not None:
                raise ValueError("DecisionInputs owns gap_input_dir; pass it on the contract")
            if any(value is not None for value in (df_exec, current_prices, lake, snapshot)):
                raise ValueError(
                    "DecisionInputs cannot be combined with df_exec, current_prices, lake, or snapshot"
                )
            input_date = inputs.trade_date.strftime("%Y-%m-%d")
            if trade_date is not None and normalize_jst_date(trade_date) != inputs.trade_date:
                raise ValueError("trade_date does not match DecisionInputs.known.trade_date")
            trade_date = input_date
            use_file_cache = inputs.use_file_cache
        if trade_date is None:
            raise ValueError("trade_date is required when DecisionInputs is not supplied")
        if gap_input_dir is not None:
            gap_input_dir = Path(gap_input_dir)
        if inputs is None and (df_exec is not None or lake is not None):
            if lake is not None and df_exec is not None:
                raise ValueError("Pass one historical source: df_exec or lake")
            if self._blpx_model is not None and current_prices is None and snapshot is None and lake is None:
                raise ValueError("current_prices is required for on-demand V2 decision.")
            if lake is None:
                assert df_exec is not None
                lake = PITDataLake(df_exec)
            inputs = lake.build_decision_inputs(
                trade_date, snapshot=snapshot,
                current_prices=None if snapshot is not None else current_prices,
                gap_input_dir=gap_input_dir, use_file_cache=use_file_cache,
                source="public_model_adapter",
            )
        if inputs is not None:
            validate_production_decision_inputs(inputs)
        return _v2_decide(
            self, trade_date=trade_date, gap_input_dir=gap_input_dir,
            overlay_enabled=overlay_enabled, use_file_cache=use_file_cache, inputs=inputs,
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

    def _compute_ondemand(
        self,
        trade_date: str,
        df_exec: pd.DataFrame,
        current_prices: dict[str, float],
        *,
        horizon: int = 1,
        snapshot: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _v2_compute_ondemand(
            self,
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=horizon,
            snapshot=snapshot,
        )

    def _multi_horizon_scores(
        self,
        trade_date: str,
        df_exec: pd.DataFrame,
        current_prices: dict[str, float],
        use_file_cache: bool = True,
        snapshot: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _v2_multi_horizon_scores(
            self,
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            use_file_cache=use_file_cache,
            snapshot=snapshot,
        )

    def _apply_overlay(
        self,
        result: PortfolioDecision,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        overlay_enabled: bool,
    ) -> PortfolioDecision:
        return _v2_apply_overlay(self, result, trade_date, df_exec, overlay_enabled)

    @staticmethod
    def _resolve_current_index(df_exec: pd.DataFrame, trade_date: str) -> int:
        return _v2_resolve_current_index(df_exec, trade_date)
