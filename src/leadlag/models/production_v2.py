"""Production v2 portfolio construction module.

Public API: parse_run_config, generate_v2_production_portfolio,
load_pit_ir_history, ProductionV2Model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config import safe_config_copy
from leadlag.config.schemas import ProductionV2RunConfig, _map_flat_to_nested
from leadlag.core.macro import download_macro_prices
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
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

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_GROSS",
    "COST_BPS_PER_GROSS",
    "LONG_COUNT",
    "SHORT_COUNT",
    "ProductionV2Model",
    "VERSION",
    "download_macro_prices",
    "generate_v2_production_portfolio",
    "generate_v2_production_portfolio_from_distribution",
    "load_pit_ir_history",
    "parse_run_config",
    "_build_current_prices_from_df_exec",
]

# Default constants (mirror ProductionV2RunConfig Pydantic defaults).
BASELINE_GROSS = 2.0
COST_BPS_PER_GROSS = 10.0
LONG_COUNT = 5
SHORT_COUNT = 5


def parse_run_config(cfg: dict) -> ProductionV2RunConfig:
    """Convert a raw (possibly flat) YAML cfg dict to a validated ``ProductionV2RunConfig``."""
    cfg = safe_config_copy(cfg) or {}
    mapped = _map_flat_to_nested(cfg)
    return ProductionV2RunConfig(**mapped)


def generate_v2_production_portfolio(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: ProductionV2RunConfig | dict,
) -> PortfolioDecision:
    """Backward-compatible wrapper around ``ProductionV2Model.decide`` (overlay off)."""
    cfg = safe_config_copy(cfg)
    run_cfg = cfg if isinstance(cfg, ProductionV2RunConfig) else parse_run_config(cfg)

    v2_model = ProductionV2Model(run_cfg, blpx_model=None, overlay_model=None)
    return v2_model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=None,
        current_prices=None,
        overlay_enabled=False,
        use_file_cache=True,
    )


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
        trade_date: str,
        gap_input_dir: str | Path | None = None,
        df_exec: pd.DataFrame | None = None,
        current_prices: dict[str, float] | None = None,
        overlay_enabled: bool = True,
        use_file_cache: bool = True,
        lake: PITDataLake | None = None,
        snapshot: MarketSnapshot | None = None,
    ) -> PortfolioDecision:
        if gap_input_dir is not None:
            gap_input_dir = Path(gap_input_dir)
        self._current_gap_input_dir = gap_input_dir
        return _v2_decide(
            self,
            trade_date=trade_date,
            gap_input_dir=gap_input_dir,
            df_exec=df_exec,
            current_prices=current_prices,
            overlay_enabled=overlay_enabled,
            use_file_cache=use_file_cache,
            lake=lake,
            snapshot=snapshot,
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
