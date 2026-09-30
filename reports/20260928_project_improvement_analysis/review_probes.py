"""Reproduce this review's small probes without APIs or production writes.

Run from the repository with its existing Python environment and an outer
process timeout. Only the JSON result beside this script is persisted.
Credential probes use synthetic values in a temporary registry.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.schemas import AppConfig, KabuApiConfig, TachibanaApiConfig
from leadlag.execution.config import build_app_config_from_dict, load_config_from_yaml
from leadlag.execution.runtime_manifest import _git_provenance
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from research.experiment_utils import _extract_metrics, record_backtest_experiment
from research.scripts.experiments.experiment_stage6_overlay_ablation_20260923 import _stats


def main() -> None:
    # Resolve the actual local configuration, then select only public fields.
    config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    v2 = config.v2
    public_config = {
        "ml_overlay_enabled": v2.ml_overlay_enabled,
        "ml_overlay_model_dir": v2.ml_overlay_model_dir,
        "gap_distribution_dir": config.gap_distribution_dir,
        "mh_blend_enabled": v2.mh_blend_enabled,
        "mh_horizons": v2.mh_horizons,
        "mh_weights": v2.mh_weights,
        "fallback_on_audit_failure": v2.fallback_on_audit_failure,
        "ondemand_fallback_enabled": v2.ondemand_fallback_enabled,
        "shadow_ondemand_validation": v2.shadow_ondemand_validation,
        "costs": v2.costs.model_dump(mode="json"),
        "risk": config.risk.model_dump(mode="json"),
        "strategy_side_leverage": config.strategy.side_leverage,
    }

    # No broker clients are instantiated. Isolate synthetic config from the
    # machine's provider selection, restoring the environment afterwards.
    previous_provider = os.environ.get("BROKER_PROVIDER")
    os.environ["BROKER_PROVIDER"] = "dry_run"
    config_probes = {}
    try:
        for name, payload in {
            "risk_typo": {"risk": {"var_stpo": 0.005}},
            "overlay_typo": {"ml_order_overlay": {"enabeld": True}},
            "slippage_typo": {"costs": {"slippage_bps_per_sdie": 999.0}},
        }.items():
            resolved = build_app_config_from_dict(payload, strict=True)
            config_probes[name] = {
                "raised": False,
                "var_stop": resolved.risk.var_stop,
                "ml_overlay_enabled": resolved.v2.ml_overlay_enabled,
                "slippage_bps_per_side": resolved.v2.costs.slippage_bps_per_side,
            }
        with TemporaryDirectory(prefix="leadlag_missing_config_", dir="/tmp") as temporary:
            load_config_from_yaml(Path(temporary) / "absent.yaml", strict=True)
            config_probes["explicit_missing_path"] = {"raised": False}
    finally:
        if previous_provider is None:
            os.environ.pop("BROKER_PROVIDER", None)
        else:
            os.environ["BROKER_PROVIDER"] = previous_provider

    synthetic = AppConfig(
        kabu=KabuApiConfig(api_token="SYNTHETIC_TOKEN", api_password="SYNTHETIC_PASSWORD"),
        tachibana=TachibanaApiConfig(second_password="SYNTHETIC_SECOND_PASSWORD"),
    )
    with TemporaryDirectory(prefix="leadlag_registry_review_", dir="/tmp") as temporary:
        registry_path = Path(temporary) / "registry.jsonl"
        record_backtest_experiment(
            name="synthetic_credential_serialization_probe",
            hypothesis="Check serialization using synthetic values only",
            app_config=synthetic,
            registry_path=registry_path,
        )
        persisted = registry_path.read_text()
        secret_probe = {
            key: value in persisted
            for key, value in {
                "token_persisted": "SYNTHETIC_TOKEN",
                "password_persisted": "SYNTHETIC_PASSWORD",
                "second_password_persisted": "SYNTHETIC_SECOND_PASSWORD",
            }.items()
        }

    with TemporaryDirectory(prefix="leadlag_git_review_", dir="/tmp") as temporary:
        repository = Path(temporary)
        subprocess.run(["git", "init", "-q", str(repository)], check=True, timeout=5)
        untracked = repository / "untracked.py"
        untracked.write_text("value = 1\n")
        first = _git_provenance(repository)
        untracked.write_text("value = 2\n")
        second = _git_provenance(repository)
        git_probe = {
            "untracked_content_changed": True,
            "dirty_diff_hash_changed": first["dirty_diff_hash"] != second["dirty_diff_hash"],
        }

    missing = _extract_metrics({"daily_returns": pd.Series([0.01, np.nan, -0.02, 0.03, 0.005])})
    first_loss = pd.Series([-0.1, 0.0])
    result = {
        "resolved_public_configuration": public_config,
        "strict_configuration_probes": config_probes,
        "synthetic_credentials": secret_probe,
        "untracked_code_provenance": git_probe,
        "metric_probes": {
            "nan_returns": {
                "net_sharpe": missing["net_sharpe"],
                "n_observations": missing["n_observations"],
                "saved_returns_count": len(missing["returns"]),
                "max_dd_is_finite": bool(np.isfinite(missing["max_dd"])),
                "total_return_is_finite": bool(np.isfinite(missing["total_return"])),
            },
            "initial_loss": {
                "stage6_max_drawdown": _stats(first_loss)["max_drawdown"],
                "canonical_max_drawdown": calculate_metrics(first_loss, spec=MetricsSpec())["MDD"],
            },
            "annualization_ratio_252_over_245": float(np.sqrt(252 / 245)),
            "stage5_sharpe252": 5.163636078955636,
            "same_series_sharpe245": float(5.163636078955636 * np.sqrt(245 / 252)),
        },
        "fixed_path_cost_diagnostic": {
            "stage8_linear_zero_sum_return_bps_per_side": 5 + 1.964607 / (0.778427 / 5),
            "assumptions": "Saved weights/gross fixed; model financing/borrow/reverse fixed; sum of daily returns, not compound wealth; no fills/impact/lot model.",
        },
    }
    destination = Path(__file__).with_name("probe_results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
