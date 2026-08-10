"""Historical daily return cache for VaR/ES risk checks.

This module was split from ``leadlag.data.market_data_cache`` to remove the
reverse dependency of the data layer on the execution layer. It runs the V2
backtest only when a cached return series is not already available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution.config import build_app_config_from_dict
from leadlag.utils.threading import run_with_timeout

logger = logging.getLogger(__name__)


def get_hist_returns_for_risk(
    strategy: Any,
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
) -> pd.Series:
    """Efficiently get historical daily returns for VaR/ES risk checks.

    Uses an SQLite cache if available, otherwise runs the V2 full backtest and
    caches the result. The ``strategy`` argument is kept for backward
    compatibility but is no longer used.
    """
    cache_dir = Path(output_root) / ".cache"
    returns_store = SqliteCacheStore(cache_dir / "daily_returns.sqlite")

    cached = returns_store.get("daily_returns")
    if cached is not None:
        hist_returns = cached["daily_return"]
        hist_returns = hist_returns[hist_returns.index < trade_date]
        logger.info("Loaded %d cached daily returns for VaR/ES", len(hist_returns))
        return hist_returns

    logger.info("No return cache found; running V2 full backtest for VaR/ES...")
    df_exec = load_df_exec_from_local_cache()

    # Resolve production config (the canonical V2 source of truth)
    project_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        resolved_cfg_path = project_root / "configs" / "production" / "production.yaml"
    else:
        resolved_cfg_path = Path(config_path)
        if not resolved_cfg_path.is_absolute():
            resolved_cfg_path = project_root / resolved_cfg_path

    with open(resolved_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Allow the caller-provided config to override start date and slippage.
    start_date = getattr(config, "start_date", "2015-01-05")
    slippage_bps = getattr(config, "slippage_bps", None)
    if slippage_bps is not None:
        cfg.setdefault("costs", {})["slippage_bps_per_side"] = float(slippage_bps)

    app_config = build_app_config_from_dict(cfg)

    if gap_input_dir is None:
        gap_input_dir = app_config.gap_distribution_dir
    gap_dir: Path | None = None
    if gap_input_dir:
        gap_dir = Path(gap_input_dir)
        if not gap_dir.is_absolute():
            gap_dir = project_root / gap_dir
        if not gap_dir.exists():
            logger.warning(
                "Gap input dir not found: %s. V2 VaR/ES history will fall back to flat positions.",
                gap_dir,
            )
            gap_dir = None

    # Local import to avoid a module-level cycle; BacktestEngine lives in execution.
    from leadlag.execution.backtester import BacktestEngine

    def _run_backtest_for_risk() -> dict[str, Any]:
        return BacktestEngine.run_v2_backtest(
            cfg=app_config,
            gap_input_dir=gap_dir,
            df_exec=df_exec,
            start_date=start_date,
            n_jobs=1,
        )

    # The full V2 backtest can take longer when no cache is available.
    # Allow the caller to override the timeout (in seconds); default 5 minutes.
    if isinstance(config, dict):
        timeout = config.get("var_history_timeout", 300)
    else:
        timeout = getattr(config, "var_history_timeout", 300)
    timeout = max(1, int(timeout))

    try:
        out_res = run_with_timeout(
            _run_backtest_for_risk,
            timeout=timeout,
            label="get_hist_returns_for_risk V2 backtest",
        )
    except TimeoutError:
        logger.warning(
            "V2 backtest for VaR/ES timed out after %d seconds; returning empty series.",
            timeout,
        )
        return pd.Series(dtype=float)
    hist_results = pd.DataFrame(
        {"daily_return": out_res["daily_returns"]},
        index=out_res["daily_returns"].index,
    )

    returns_store.set("daily_returns", hist_results)

    return pd.Series(
        hist_results.loc[
            hist_results.index < trade_date,
            "daily_return",
        ]
    )
