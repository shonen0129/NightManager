"""Offline before/after numerical replay against the frozen regression dataset.

Run with the chosen source tree on PYTHONPATH. No broker or network is used.
This comparison explicitly excludes production ML/macro/ADR availability.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data import intraday_inputs, macro
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.runner.model_factory import build_v2_model_bundle
from leadlag.utils.gap_matrix_io import save_gap_matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    baseline = root / "tests/regression/baselines"
    frame = pd.read_csv(baseline / "df_exec_20260814.csv.gz", index_col=0, parse_dates=True)
    # Publish the same fixture arrays with the supported manifest contract.
    # Legacy-manifest rejection is a separate reproduced bug, not a numerical
    # refactor baseline. These files live only in this replay's temporary store.
    owner = tempfile.TemporaryDirectory(prefix="leadlag-replay-gap-")
    cache_dir = Path(owner.name)
    for horizon in (1, 3, 5):
        suffix = "" if horizon == 1 else f"_h{horizon}"
        mu = np.load(baseline / f"matrices/mu_gap{suffix}_20260814.npy")
        omega = np.load(baseline / f"matrices/omega_gap{suffix}_20260814.npy")
        kwargs = {} if horizon == 1 else {"mu_pattern": "matrices/mu_gap_h{h}_{date}.npy",
                                         "omega_pattern": "matrices/omega_gap_h{h}_{date}.npy",
                                         "pattern_kwargs": {"h": horizon}}
        assert save_gap_matrices(cache_dir, "2026-08-14", mu, omega,
                                 metadata={"sig_date": "2026-08-13", "trade_date": "2026-08-14", "horizon": horizon},
                                 **kwargs)
    macro.load_macro_prices = lambda **_: None
    intraday_inputs.build_open_910_returns = lambda df, tickers: pd.DataFrame(0.0, index=df.index, columns=tickers)
    intraday_inputs.build_5m_910_prices = lambda *a, **kw: pd.DataFrame()
    config = load_config_from_yaml(root / "configs/production/production.yaml", strict=True)
    config = config.model_copy(update={"v2": config.v2.model_copy(update={
        "ml_overlay_enabled": False, "macro_kappa_enabled": False, "macro_direction_enabled": False,
        "cs_overlay_enabled": False, "gap_input_dir": cache_dir,
    })})
    model = build_v2_model_bundle(config).decision_model
    date = "2026-08-14"
    snapshot = PITDataLake(frame).get_snapshot(date)
    prices = dict(snapshot.current_prices)
    result = {}
    for horizon in (1, 3, 5):
        for use_cache in (True, False):
            mu, omega = model.compute_distribution(date, frame, prices, horizon=horizon,
                                                   snapshot=snapshot, use_file_cache=use_cache)
            if use_cache:
                suffix = "" if horizon == 1 else f"_h{horizon}"
                np.testing.assert_array_equal(mu, np.load(baseline / f"matrices/mu_gap{suffix}_20260814.npy"))
            result[f"h{horizon}_{'cache' if use_cache else 'ondemand'}"] = {
                "mu": mu.tolist(), "omega": omega.tolist(),
            }
    for enabled in (False, True):
        cfg = config.model_copy(update={"v2": config.v2.model_copy(update={"mh_blend_enabled": enabled})})
        decision = build_v2_model_bundle(cfg).decision_model.decide(
            inputs=PITDataLake(frame).build_decision_inputs(date, gap_input_dir=cache_dir),
            overlay_enabled=False,
        )
        assert np.isfinite(decision.w_final).all()
        assert np.abs(decision.w_final).sum() > 0, "A flat-only comparison is insufficient"
        assert len(decision.w_final) == len(JP_TICKERS)
        result[f"decision_mh_{enabled}"] = {
            "weights": decision.w_final.tolist(), "scores": decision.scores.tolist(),
            "pit": decision.pit_binning, "fallback": decision.fallback,
        }
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    owner.cleanup()


if __name__ == "__main__":
    main()
