"""Compatibility boundary for the production ML order-decision overlay.

The fitted container stays in this module so existing pickle files continue to
resolve ``leadlag.models.ml_order_overlay.MLOrderOverlayModel``.  Feature
construction, artifact persistence, production inference, and research
training are implemented in separate modules and re-exported here only for
callers that still use the historical import path.
"""

from __future__ import annotations

import logging
import os  # Compatibility: existing tests patch overlay.os.replace during publication.
from dataclasses import dataclass, field
from typing import Any

from leadlag.data.adr_features import DEFAULT_ADR_FEATURES_PATH

logger = logging.getLogger(__name__)

TRADING_DAYS = 245
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0
DEFAULT_LGBM_KWARGS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "num_leaves": 20,
    "learning_rate": 0.05,
    "min_child_samples": 300,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class MLOrderOverlayModel:
    """Container for a fitted LightGBM overlay model.

    Do not move this class without an artifact migration.  Pickle resolution
    of this fully-qualified class path is a production compatibility contract.
    """

    lgbm: Any
    cont_cols: list[str]
    target_std: float
    use_ticker: bool
    use_classification: bool
    per_ticker_interactions: bool
    p_trade_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Historical names remain importable while the dedicated modules are canonical.
from leadlag.models.ml_overlay_artifact import (  # noqa: E402
    _normalize_overlay_date,
    _validate_overlay_provenance,
    datetime_now_for_artifact,
    load_overlay_model,
    save_overlay_model,
)
from leadlag.models.ml_overlay_features import (  # noqa: E402
    _build_ticker_features,
    _precompute_market_vol,
    _predict_p_trade,
    _recompute_w_pre,
    _safe,
    _sigmoid,
    overlay_continuous_columns,
)
from leadlag.models.ml_overlay_inference import (  # noqa: E402
    apply_overlay,
    generate_v2_production_portfolio_with_overlay,
)

_COMPAT_OS_MODULE = os


def train_overlay_model(*args: Any, **kwargs: Any) -> MLOrderOverlayModel:
    """Reject the retired production training entry point with migration guidance."""
    del args, kwargs
    raise RuntimeError(
        "Overlay training is a research operation. Import "
        "research.experiments.ml_overlay_training in the research environment."
    )


__all__ = [
    "DEFAULT_ADR_FEATURES_PATH",
    "DEFAULT_LGBM_KWARGS",
    "MLOrderOverlayModel",
    "ROUND_TRIP_COST",
    "SLIPPAGE_BPS_PER_SIDE",
    "TRADING_DAYS",
    "_build_ticker_features",
    "_normalize_overlay_date",
    "_predict_p_trade",
    "_precompute_market_vol",
    "_recompute_w_pre",
    "_safe",
    "_sigmoid",
    "_validate_overlay_provenance",
    "apply_overlay",
    "datetime_now_for_artifact",
    "generate_v2_production_portfolio_with_overlay",
    "load_overlay_model",
    "overlay_continuous_columns",
    "save_overlay_model",
    "train_overlay_model",
]
