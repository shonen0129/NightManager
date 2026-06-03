"""Configuration values for research/backtest scripts only."""

import os

from results_format import create_results_output_dir, get_default_results_root

from config import DEFAULT_START_DATE


def create_timestamped_output_dir(run_name="backtest", base_dir=None):
    """Create and return a timestamped results directory for one run."""
    root = (
        os.path.abspath(base_dir)
        if base_dir is not None
        else get_default_results_root()
    )
    return create_results_output_dir(
        run_name=run_name,
        output_root=root,
        manifest_extra={"entry_point": "backtest_config.create_timestamped_output_dir"},
    )


LOGIC_DIFF_BASELINE_PARAMS = {
    "K": 3,
    "lambda_reg": 0.75,
    "q": 0.3,
    "weight_mode": "equal",
    "dispersion_filter": False,
    "v3_mode": "static",
    "ewma_half_life": None,
    "lambda_lw": 0.0,
    "lw_target": "identity",
    "corr_window": 60,
    "include_v4_prior": False,
    "signal_mode": "baseline",
}

LOGIC_DIFF_TOP1_PARAMS = {
    "K": 3,
    "lambda_reg": 0.75,
    "q": 0.3,
    "weight_mode": "signal",
    "dispersion_filter": True,
    "v3_mode": "static",
    "ewma_half_life": 45,
    "lambda_lw": 0.5,
    "lw_target": "equicorrelation",
    "corr_window": 60,
    "include_v4_prior": False,
}

LOGIC_DIFF_INCREMENTAL_STEPS = [
    {"label": "S1 +EWMA45", "update": {"ewma_half_life": 45}},
    {"label": "S2 +SignalWeight", "update": {"weight_mode": "signal"}},
    {
        "label": "S3 +DispersionFilter",
        "update": {"dispersion_filter": True},
    },
    {
        "label": "S4 +TwoStageShrink",
        "update": {"lambda_lw": 0.5, "lw_target": "equicorrelation"},
    },
    {"label": "S5 +FXPrior(v4)", "update": {"include_v4_prior": True}},
    {"label": "S6 +K4", "update": {"K": 4}},
]

LOGIC_DIFF_SINGLE_FACTOR_CHANGES = [
    {"label": "F1 EWMA45 only", "update": {"ewma_half_life": 45}},
    {"label": "F2 SignalWeight only", "update": {"weight_mode": "signal"}},
    {
        "label": "F3 DispersionFilter only",
        "update": {"dispersion_filter": True},
    },
    {
        "label": "F4 TwoStageShrink only",
        "update": {"lambda_lw": 0.5, "lw_target": "equicorrelation"},
    },
    {"label": "F5 FXPrior(v4) only", "update": {"include_v4_prior": True}},
    {"label": "F6 K4 only", "update": {"K": 4}},
    {
        "label": "F7 FXPrior(v4)+K4 only",
        "update": {"include_v4_prior": True, "K": 4},
    },
]

LOGIC_DIFF_EXHAUSTIVE_OPTIONS = {
    "ewma_half_life": [None, 45],
    "weight_mode": ["equal", "signal"],
    "dispersion_filter": [False, True],
    "two_stage": [False, True],
    "include_v4_prior": [False, True],
    "K": [3, 4],
}

LOGIC_DIFF_EXHAUSTIVE_BASE_PARAMS = {
    "lambda_reg": 0.75,
    "q": 0.3,
    "v3_mode": "static",
    "corr_window": 60,
}

SIGNIFICANCE_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "trading_days": 245,
    "bootstrap_samples": 2000,
    "bootstrap_block": 20,
    "bootstrap_seed": 42,
    "reality_check_seed": 7,
    "periods": [
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-now", "2023-01-01", None),
    ],
}

TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "trading_days": 245,
    "hac_lags": 10,
    "bootstrap_samples": 2000,
    "bootstrap_block": 20,
    "bootstrap_seed": 42,
}

EWMA_STATIC_GRID_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "half_lives": [10, 15, 20, 30, 45],
    "weight_mode": "signal",
    "dispersion_filter": True,
    "v3_mode": "static",
}

MULTI_WINDOW_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "baseline_window": 60,
    "ensemble_windows": [20, 60, 120],
    "combine_mode": "performance_weighted",
    "performance_lookback": 60,
    "strategy_params": {
        "K": 3,
        "lambda_reg": 0.75,
        "q": 0.3,
        "weight_mode": "signal",
        "dispersion_filter": True,
        "v3_mode": "static",
        "ewma_half_life": 45,
        "lambda_lw": 0.5,
        "lw_target": "equicorrelation",
        "include_v4_prior": False,
    },
}

TWO_STAGE_SHRINK_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "lambda_reg": 0.75,
    "lambda_grid": [round(i * 0.05, 2) for i in range(21)],
    "lw_targets": ["identity", "equicorrelation"],
    "strategy_params": {
        "K": 3,
        "q": 0.3,
        "weight_mode": "signal",
        "dispersion_filter": True,
        "v3_mode": "static",
        "ewma_half_life": 45,
        "include_v4_prior": False,
    },
}

GAP_RESIDUAL_COMPARE_CONFIG = {
    "start_date": DEFAULT_START_DATE,
    "gap_open_coef": 1.0,
    "baseline_params": LOGIC_DIFF_TOP1_PARAMS,
}
