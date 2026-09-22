"""Offline review probe: training and production must see the same base score."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.pit_lake import PITDataLake
from leadlag.data.adr_features import load_adr_features, validate_adr_features
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.ml_overlay_features import _build_ticker_features, _precompute_market_vol, overlay_continuous_columns
from leadlag.models.v2.gap_io import _extract_horizon_snapshot_inputs
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity
from research.experiments.ml_overlay_training import _collect_training_data
from research.experiments import ml_overlay_training as training


def main():
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml").v2
    # Isolate the multi-horizon difference. Keep production horizons/weights.
    cfg = cfg.model_copy(update={"macro_kappa_enabled": False,
                                 "macro_direction_enabled": False,
                                 "cs_overlay_enabled": True})
    dates = pd.bdate_range("2026-08-10", periods=25)
    data = {"sig_date": (dates - pd.offsets.BDay(1)), "topix_night_return": 0.0}
    data.update({f"us_cc_{t}": 0.0 for t in US_TICKERS})
    for t in JP_TICKERS:
        for name, value in (("jp_gap", 0.0), ("jp_beta", 1.0),
                            ("jp_open_trade", 100.0), ("jp_close_sig", 100.0),
                            ("jp_oc", 0.01)):
            data[f"{name}_{t}"] = value
    frame = pd.DataFrame(data, index=dates)
    open910 = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)
    vol = _precompute_market_vol(frame)
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
        features = _build_ticker_features(frame, production, dates[-1], vol, snapshot=snap)
        columns = overlay_continuous_columns()
        np.testing.assert_allclose(train[columns], features[columns], rtol=0, atol=0, equal_nan=True)
        print("PASS: all 17 continuous features match on the normalized complete fixture")
        interaction_rows = _collect_training_data(
            dates[-1:], frame, np.full((len(dates), len(JP_TICKERS)), 0.01),
            path, cfg, vol, open_910_returns=open910, per_ticker_interactions=True,
        )
        interaction_features = _build_ticker_features(
            frame, production, dates[-1], vol, snapshot=snap, per_ticker_interactions=True,
        )
        columns = overlay_continuous_columns(True)
        np.testing.assert_allclose(interaction_rows[columns], interaction_features[columns], rtol=0, atol=0, equal_nan=True)
        print(f"PASS: all {len(columns)} continuous features with ticker interactions match")

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

        # A future common timestamp must remain relevant when a different
        # feature has a per-date timestamp. Neither mapping duplicates a key.
        try:
            lake.build_decision_inputs(
                date, gap_input_dir=path, open_910_returns=open910, source="review",
                rank_reversal_signals=rank,
                historical_observed_at={"rank_reversal_signals": date + " 09:20"},
            )
        except ValueError as exc:
            assert "historical observed_at" in str(exc)
            print("PASS: common future timestamp alone is rejected")
        else:
            raise AssertionError("common future timestamp alone was accepted")
        mixed = lake.build_decision_inputs(
            date, gap_input_dir=path, open_910_returns=open910, source="review",
            rank_reversal_signals=rank,
            historical_observed_at={"rank_reversal_signals": date + " 09:20"},
            historical_observed_at_by_date={date: {"macro_prices": date + " 09:00"}},
        )
        result = ProductionV2Model(cfg, blpx_model=object()).decide(inputs=mixed, overlay_enabled=False)
        score_delta = float(np.max(np.abs(result.scores - production.scores)))
        weight_delta = float(np.max(np.abs(result.w_final - production.w_final)))
        assert not result.fallback.get("audit_failure")
        assert score_delta > 0 and weight_delta > 0
        print(json.dumps({"finding": "M1", "common": dict(mixed.historical.observed_at),
            "selected": dict(mixed.historical.observed_at_for(date)),
            "score_delta": score_delta, "weight_delta": weight_delta,
            "fallback": result.fallback}, indent=2, default=str))

        # Load an actual temporary ADR artifact via the normal loader. No
        # market-data source is fetched, and no fitted artifact is published.
        adr = pd.DataFrame(0.03, index=dates, columns=[f"adr_{t}" for t in JP_TICKERS])
        adr.index = adr.index.tz_localize("Asia/Tokyo").tz_convert("UTC")
        adr_path = path / "adr.pkl"
        adr.to_pickle(adr_path)
        loaded = load_adr_features(adr_path)
        assert loaded is not None
        normalized = validate_adr_features(loaded, date)
        assert normalized is not None and normalized.index.tz is None
        kwargs = dict(df_exec=frame, gap_input_dir=path, run_cfg=cfg,
            train_start=str(dates[0].date()), train_end=date, output_dir=path / "unpublished")
        outcomes = {}
        for name, supplied in (("UTC", loaded), ("JST_normalized", normalized)):
            with patch.object(training, "build_open_910_returns", return_value=open910), \
                 patch.object(training, "load_adr_features", return_value=supplied), \
                 patch.object(training, "_collect_training_data", return_value=pd.DataFrame()) as collect:
                try:
                    training._train_overlay_model_impl(**kwargs)
                except (TypeError, ValueError) as exc:
                    outcomes[name] = {"error": str(exc), "type": type(exc).__name__,
                                      "reached_collector": collect.called}
        assert outcomes["UTC"]["type"] == "TypeError"
        assert not outcomes["UTC"]["reached_collector"]
        assert outcomes["JST_normalized"]["reached_collector"]
        print(json.dumps({"finding": "M2", "outcomes": outcomes}, indent=2))


if __name__ == "__main__":
    main()
