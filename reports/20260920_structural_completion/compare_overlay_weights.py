"""Compare one live-style decision and one canonical backtest decision."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root
from leadlag.data import adr_features as adr_data
from leadlag.data import macro as macro_data
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.runner.model_factory import build_v2_model_bundle


def main() -> None:
    root = project_root()
    date = pd.Timestamp("2026-08-14")
    gap_dir = root / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite"
    artifact = root / "var/results/20260920_structural_completion/ml_overlay_retrained"
    config = load_config_from_yaml(root / "configs/production/production.yaml", strict=True)
    df_exec = load_df_exec_from_local_cache()
    bundle = build_v2_model_bundle(config, overlay_model_dir=artifact, clear_blpx_cache=True)
    lake = PITDataLake(df_exec)
    as_of = date + pd.Timedelta(hours=9, minutes=10)
    snap = lake.get_snapshot(as_of)
    open_910 = build_open_910_returns(df_exec, JP_TICKERS)
    rank = load_rank_reversal_frame(gap_dir, [date], file_pattern=config.v2.cs_rank_reversal_file_pattern)
    pit_ir, _pit_alerts, pit_dates = load_pit_ir_history(gap_dir, date.strftime("%Y-%m-%d"))
    try:
        adr = adr_data.load_adr_features()
    except Exception:
        adr = None
    try:
        macro = macro_data.load_macro_prices(
            start=df_exec.index.min().strftime("%Y-%m-%d"),
            end=(df_exec.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            period="max",
        )
    except Exception:
        macro = None
    historical = HistoricalInputs(
        df_exec,
        source="r5_compare",
        open_910_returns=open_910,
        macro_prices=macro,
        adr_features_frame=adr,
        rank_reversal_signals=rank,
        pit_ir_history=pit_ir,
        pit_history_trade_dates=pit_dates,
    )
    inputs = DecisionInputs(
        known=snap.to_known_inputs(
            source="backtest_pit",
            observed_at={
                "us_returns": f"{date.date()} 09:00",
                "jp_gap_returns": f"{date.date()} 09:10",
                "jp_betas": f"{date.date()} 09:10",
                "topix_night_return": f"{date.date()} 09:10",
                "current_prices": f"{date.date()} 09:10",
                "prev_closes": f"{date.date()} 09:10",
            },
        ),
        historical=historical,
        gap_input_dir=gap_dir,
    )
    live_style = bundle.decision_model.decide(inputs=inputs, overlay_enabled=True)
    backtest = BacktestEngine.run_v2_backtest(
        config,
        gap_dir,
        df_exec,
        start_date=date.strftime("%Y-%m-%d"),
        end_date=date.strftime("%Y-%m-%d"),
        n_jobs=1,
        overlay_model=bundle.overlay_model,
        overlay_model_dir=artifact,
    )
    backtest_frame = backtest["weights"]
    if date not in backtest_frame.index:
        raise RuntimeError(f"backtest date missing; returned range {backtest_frame.index.min()}..{backtest_frame.index.max()}")
    backtest_weights = backtest_frame.loc[date].to_numpy(dtype=float)
    payload = {
        "date": date.strftime("%Y-%m-%d"),
        "artifact": str(artifact),
        "overlay_version": getattr(bundle.overlay_model, "artifact_version", None),
        "max_abs_weight_diff": float(np.max(np.abs(live_style.w_final - backtest_weights))),
        "l2_weight_diff": float(np.linalg.norm(live_style.w_final - backtest_weights)),
        "live_fallback": live_style.fallback,
        "backtest_fallback": bool(backtest["daily_fallback"].loc[date]),
        "live_summary": live_style.summary,
        "backtest_summary": backtest["v2_summaries"][0] if backtest["v2_summaries"] else None,
    }
    out = root / "reports/20260920_structural_completion/evidence/r5_weights.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
