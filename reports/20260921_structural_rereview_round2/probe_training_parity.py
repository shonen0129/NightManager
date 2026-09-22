"""Offline review probe: training and production must see the same base score."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.gap_io import _extract_horizon_snapshot_inputs
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity
from research.experiments.ml_overlay_training import _collect_training_data


def main():
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml").v2
    # Isolate the multi-horizon difference. Keep production horizons/weights.
    cfg = cfg.model_copy(update={"macro_kappa_enabled": False,
                                 "macro_direction_enabled": False,
                                 "cs_overlay_enabled": True})
    dates = pd.bdate_range("2026-09-07", periods=5)
    data = {"sig_date": (dates - pd.offsets.BDay(1)), "topix_night_return": 0.0}
    data.update({f"us_cc_{t}": 0.0 for t in US_TICKERS})
    for t in JP_TICKERS:
        for name, value in (("jp_gap", 0.0), ("jp_beta", 1.0),
                            ("jp_open_trade", 100.0), ("jp_close_sig", 100.0),
                            ("jp_oc", 0.01)):
            data[f"{name}_{t}"] = value
    frame = pd.DataFrame(data, index=dates)
    open910 = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)
    vol = pd.DataFrame(0.01, index=dates, columns=JP_TICKERS)
    lake = PITDataLake(frame)
    date = dates[-1].strftime("%Y-%m-%d")
    snap = lake.get_snapshot(date + " 09:10")
    with tempfile.TemporaryDirectory(prefix="leadlag-review-training-") as directory:
        path = Path(directory)
        cfg = cfg.model_copy(update={"gap_input_dir": path})
        vectors = {1: np.linspace(-0.01, 0.01, len(JP_TICKERS)),
                   3: np.sin(np.arange(len(JP_TICKERS))) * 0.03,
                   5: np.cos(np.arange(len(JP_TICKERS))) * 0.03}
        for horizon, mu in vectors.items():
            gap_inputs = ((snap.jp_gap_returns, snap.jp_betas, snap.topix_night_return)
                          if horizon == 1 else
                          _extract_horizon_snapshot_inputs(frame, date, horizon, snap))
            metadata = {"sig_date": str(frame.loc[dates[-1], "sig_date"].date()),
                        "trade_date": date, "horizon": horizon,
                        **bundle_identity(frame, date, config=cfg,
                                          open_910_returns=open910,
                                          gap_inputs=gap_inputs, horizon=horizon)}
            assert save_gap_matrices(path, date, mu, np.eye(len(JP_TICKERS)) * 0.001,
                mu_pattern=cfg.mu_file_pattern if horizon == 1 else cfg.mh_mu_file_pattern_h,
                omega_pattern=cfg.omega_file_pattern if horizon == 1 else cfg.mh_omega_file_pattern_h,
                pattern_kwargs=None if horizon == 1 else {"h": horizon}, metadata=metadata)
        inputs = lake.build_decision_inputs(date, gap_input_dir=path,
                                            open_910_returns=open910, source="review")
        # A marker is sufficient: all three distributions must resolve from cache.
        production = ProductionV2Model(cfg, blpx_model=object()).decide(
            inputs=inputs, overlay_enabled=False)
        assert not production.fallback.get("gap_data_missing")
        assert not production.fallback.get("audit_failure")
        train = _collect_training_data(dates[-1:], frame,
            np.full((len(dates), len(JP_TICKERS)), 0.01), path, cfg, vol,
            open_910_returns=open910)
        assert len(train) == len(JP_TICKERS)
        output = {"horizons": cfg.mh_horizons, "weights": cfg.mh_weights,
                  "training_scores": train["score"].tolist(),
                  "production_scores": production.scores.tolist(),
                  "max_score_difference": float(np.max(np.abs(train["score"].values - production.scores))),
                  "side_disagreements": int(np.sum(np.sign(train["score"].values) != np.sign(production.scores))),
                  "production_fallback": production.fallback}
        print(json.dumps(output, indent=2, default=str))
        assert output["max_score_difference"] < 1e-12
        assert output["side_disagreements"] == 0
        print("FIXED: identical typed cached inputs yield identical training/production features")

        rank = pd.DataFrame([np.sin(np.arange(len(JP_TICKERS)))],
                            index=dates[-1:], columns=JP_TICKERS)
        try:
            lake.build_decision_inputs(
                date, gap_input_dir=path, open_910_returns=open910, source="review",
                rank_reversal_signals=rank,
                historical_observed_at_by_date={
                    date: {"rank_reversal_signals": date + " 09:20"}
                },
            )
        except ValueError as exc:
            assert "historical observed_at" in str(exc)
            print(f"FIXED: future auxiliary observation rejected: {exc}")
        else:
            raise AssertionError("future auxiliary observation was accepted")


if __name__ == "__main__":
    main()
