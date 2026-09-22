"""Replay gap and VaR with frozen real inputs; broker operations are absent."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from leadlag.config.schemas import AppConfig
from leadlag.data.horizon_returns import compute_cumulative_returns
from leadlag.data.intraday_inputs import compute_jp_target_returns
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution import var_history
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_artifact import load_overlay_model
from leadlag.models.v2.distribution_source import (
    FileCacheDistributionSource,
    OnDemandDistributionSource,
)
from leadlag.runner.model_factory import build_blpx_model, build_v2_model_bundle
from leadlag.utils.gap_provenance import config_version
from research.diagnostics.gap_inputs import build_gap_historical_inputs, prepare_gap_model_inputs
from research.scripts.experiments.structural_artifact_acceptance import (
    GAP,
    REPORT,
    ROOT,
    WORK,
    load,
    resources,
    write_json,
)


def gap():
    sys.path.insert(0, str(ROOT / "tools/research"))
    from compute_gap_adjusted_distribution import (
        GapDistAccumulators,
        GapDistContext,
        _process_date_impl,
    )
    logging.getLogger().setLevel(logging.ERROR)
    frame, history, config = resources()
    app = AppConfig(v2=config)
    # The ML artifact is irrelevant to the distribution producer; only its
    # canonical BLPX dependency is constructed here.
    models = {h: build_blpx_model(app) for h in (1, 3, 5)}
    intraday = history.open_910_returns
    inputs, histories = {}, {}
    for h in models:
        histories[h] = build_gap_historical_inputs(frame, open_910_returns=intraday, horizon=h)
        targets = compute_jp_target_returns(frame, JP_TICKERS, horizon=h, open_910_returns=intraday)
        inputs[h] = prepare_gap_model_inputs(models[h], compute_cumulative_returns(frame, h),
                                              y_jp_target=targets, horizon=h, open_910_returns=intraday,
                                              historical_inputs=histories[h])
    base = inputs[1]
    output = WORK / "gap_regenerated"
    (output / "matrices").mkdir(parents=True, exist_ok=True)
    context = GapDistContext(
        df_exec=frame, model=models[1], jp_gap=base.jp_gap, jp_beta=base.jp_beta, topix_night=base.topix_night,
        jp_res_returns_p3=base.jp_res_returns_p3, v0_static=base.v0_static, c_full_p3=base.c_full_p3,
        dist_in_dir=ROOT / "var/results/distribution_diagnostics/20260708_184812", out_dir=output,
        save_daily_m=True, save_mh=True, mh_horizons=[3, 5], mh_models=models, mh_inputs=inputs,
        save_rr=False, y_jp_target=base.y_jp_target, weights_df=pd.DataFrame(columns=JP_TICKERS),
        turnover_map={}, cfg={"costs": config.costs.model_dump()}, c=.7, b=.6,
        bl_mh_enabled=config.mh_blend_enabled, bl_mh_horizons=config.mh_horizons, bl_mh_weights=config.mh_weights,
        bl_mh_mu_pattern=config.mh_mu_file_pattern_h, bl_mh_omega_pattern=config.mh_omega_file_pattern_h,
        bl_cs_overlay_enabled=False, bl_cs_overlay_weight=config.cs_overlay_weight,
        bl_rr_pattern=config.cs_rank_reversal_file_pattern, bl_long_count=config.long_count,
        bl_short_count=config.short_count, bl_minvar_enabled=config.minvar_enabled,
        bl_minvar_alpha=config.minvar_alpha, bl_baseline_gross=config.baseline_gross,
        bl_cost_bps_per_gross=config.cost_bps_per_gross, open_910_returns=intraday,
        historical_inputs=histories[1], historical_inputs_by_horizon=histories,
        bundle_config_version=config_version(config),
    )
    # Build the same V2 distribution resolver without loading the unrelated
    # rejected production overlay; no setting involved in BLPX is changed.
    from leadlag.models.production_v2 import ProductionV2Model
    model = ProductionV2Model(config, blpx_model=build_blpx_model(app))
    file_source = FileCacheDistributionSource(model, gap_input_dir=output)
    ondemand_source = OnDemandDistributionSource(model)
    lake = PITDataLake(frame)
    results = []
    for date_str in ("2026-07-24", "2026-08-06", "2026-08-14"):
        date = pd.Timestamp(date_str)
        accumulator = GapDistAccumulators(omega_gap_ticker_records={t: [] for t in JP_TICKERS})
        _process_date_impl(date, context, accumulator)
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_execution_snapshot(as_of, intraday)
        for h in (1, 3, 5):
            kwargs = dict(trade_date=date_str, df_exec=history.calculation_frame(as_of),
                          current_prices=dict(snapshot.current_prices), horizon=h, snapshot=snapshot,
                          open_910_returns=intraday, allow_implicit_io=False)
            cached, computed = file_source.resolve(**kwargs), ondemand_source.resolve(**kwargs)
            row = {"date": date_str, "horizon": h, "cache_status": cached.status.value,
                   "ondemand_status": computed.status.value, "cache_alerts": cached.alerts,
                   "ondemand_alerts": computed.alerts, "price_sources": dict(snapshot.price_sources)}
            if cached.is_available and computed.is_available:
                row["mu_max_error"] = float(np.max(np.abs(cached.mu_gap - computed.mu_gap)))
                row["omega_max_error"] = float(np.max(np.abs(cached.Omega_gap - computed.Omega_gap)))
            results.append(row)
    write_json(REPORT / "gap_path_parity.json", results)
    assert all(r.get("mu_max_error", 1) < 1e-10 and r.get("omega_max_error", 1) < 1e-10 for r in results), results
    print(json.dumps({"comparisons": len(results), "max_mu_error": max(r["mu_max_error"] for r in results),
                      "max_omega_error": max(r["omega_max_error"] for r in results)}))


def var():
    logging.getLogger().setLevel(logging.ERROR)
    frame, history, _ = resources()
    candidate_dir = WORK / "artifacts/evaluation_2025"
    overlay = load_overlay_model(candidate_dir)
    config_path = ROOT / "configs/research/structural_acceptance_20260922.yaml"
    app = load_config_from_yaml(config_path, strict=True)
    assert build_v2_model_bundle(app).overlay_enabled
    expected = load(WORK / "evaluation/2025.pkl")["candidate"]["daily_returns"]
    output = WORK / "var_replay"
    observations = []
    # Replace acquisition boundaries with the owned market inputs. The actual
    # VaR snapshot builder, deadlines, key, worker, canonical BT, SQLite write,
    # freshness check and cache read remain active. This is historical replay,
    # not evidence that today's provider inputs are fresh.
    with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame.copy()), \
         patch("leadlag.data.intraday_inputs.build_open_910_returns", return_value=history.open_910_returns), \
         patch("leadlag.data.macro.load_macro_prices", return_value=history.macro_prices), \
         patch("leadlag.data.adr_features.load_adr_features", return_value=history.adr_features), \
         patch("leadlag.data.rank_reversal.load_rank_reversal_frame", return_value=history.rank_reversal_signals):
        for label in ("cold", "warm"):
            start = time.monotonic()
            result = var_history.get_hist_returns_for_risk(
                None, {"start_date": "2025-01-01", "var_history_timeout": 600}, str(output),
                pd.Timestamp("2026-08-17"), config_path=config_path, gap_input_dir=GAP, overlay_model=overlay,
            )
            pd.testing.assert_index_equal(result.index, expected.index)
            np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), atol=1e-10, rtol=0)
            observations.append({"mode": label, "days": len(result), "elapsed_seconds": time.monotonic() - start,
                                 "max_return_error": float(np.max(np.abs(result.to_numpy() - expected.to_numpy()))),
                                 "last_date": str(result.index[-1].date()), "artifact_version": overlay.metadata["artifact_version"]})
            write_json(REPORT / "var_path_parity.json", observations)
            print(json.dumps(observations[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["gap", "var"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    {"gap": gap, "var": var}[args.stage]()
