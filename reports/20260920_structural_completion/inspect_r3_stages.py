"""Decompose Step 2 cache versus on-demand differences for two dates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root
from leadlag.data.horizon_returns import compute_cumulative_returns
from leadlag.data.intraday_inputs import (
    build_5m_910_prices,
    build_open_910_returns,
    compute_jp_target_returns,
)
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.v2.gap_io import _extract_gap_inputs
from leadlag.pipeline.gap_distribution import compute_gap_distribution, select_gap_coefficients
from leadlag.runner.model_factory import build_blpx_model, build_v2_model_bundle
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.diagnostics.gap_inputs import attach_topix_trade_returns, load_gap_execution_inputs


def _diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {"max_abs": float(np.nanmax(np.abs(d))), "l2": float(np.linalg.norm(d))}


def main() -> None:
    root = project_root()
    cache_dir = Path(os.environ.get(
        "R3_GAP_DIR",
        "var/results/20260920_structural_completion/r3_regenerated_gap/20260920_063702",
    ))
    captured_path = Path(os.environ.get(
        "R3_CAPTURE_JSON",
        "reports/20260920_structural_completion/evidence/r3_captured_gap_calls.json",
    ))
    captured = json.loads(captured_path.read_text(encoding="utf-8")) if captured_path.exists() else {"calls": []}
    frame_pickle = os.environ.get("R3_FRAME_PICKLE")
    if frame_pickle:
        df_exec = pd.read_pickle(frame_pickle)
    elif os.environ.get("R3_FRAME_SOURCE", "local") == "research":
        research_inputs = load_gap_execution_inputs()
        df_exec = attach_topix_trade_returns(research_inputs.df_exec, research_inputs.raw_data)
    else:
        df_exec = load_df_exec_from_local_cache()
    p910 = build_5m_910_prices(df_exec, JP_TICKERS)
    open910 = build_open_910_returns(df_exec, JP_TICKERS)
    lake = PITDataLake(df_exec)
    config = load_config_from_yaml(root / "configs/production/production.yaml")
    config = config.model_copy(update={"v2": config.v2.model_copy(update={"ml_overlay_enabled": False})})
    rows: list[dict[str, object]] = []
    for date in (pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-17")):
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_snapshot(as_of)
        hist = HistoricalInputs(df_exec, source="r3_stage_inspection", open_910_returns=open910)
        hist_frame = hist.calculation_frame(as_of)
        row: dict[str, object] = {"date": date.strftime("%Y-%m-%d"), "horizons": {}}
        for h in (1, 3, 5):
            full_h = compute_cumulative_returns(df_exec, h)
            hist_h = compute_cumulative_returns(hist_frame, h)
            target_full = compute_jp_target_returns(df_exec, JP_TICKERS, horizon=h, p_910_df=p910)
            target_hist = compute_jp_target_returns(hist_frame, JP_TICKERS, horizon=h, open_910_returns=open910)
            model_full = build_blpx_model(config)
            model_hist = build_v2_model_bundle(config, clear_blpx_cache=True).blpx_model
            in_full = model_full._prepare_common_inputs(full_h, horizon=h, y_jp_target=target_full, p_910_df=p910)
            in_hist = model_hist._prepare_common_inputs(
                hist_h,
                horizon=h,
                y_jp_target=target_hist,
                open_910_returns=open910,
                allow_implicit_io=False,
            )
            index_full = int(full_h.index.get_loc(date))
            index_hist = int(hist_h.index.get_loc(date))
            gap, beta, topix = _extract_gap_inputs(hist_frame, date.strftime("%Y-%m-%d"), dict(snapshot.current_prices), snapshot=snapshot)
            row_input = df_exec.loc[date]
            frame_gap = np.array([row_input.get(f"jp_gap_{tk}", 0.0) for tk in JP_TICKERS], dtype=float)
            frame_beta = np.array([row_input.get(f"jp_beta_{tk}", 0.0) for tk in JP_TICKERS], dtype=float)
            frame_topix = float(row_input.get("topix_night_return", 0.0))
            result_full = model_full.compute_blp_signal(
                in_full["jp_res_returns_p3"], index_full, v0_static=in_full["v0_static"], c_full=in_full["c_full_p3"],
                is_residual=True, return_matrices=True,
            )
            result_hist = model_hist.compute_blp_signal(
                in_hist["jp_res_returns_p3"], index_hist, v0_static=in_hist["v0_static"], c_full=in_hist["c_full_p3"],
                is_residual=True, return_matrices=True,
            )
            coef_full = select_gap_coefficients(model_full, result_full)
            coef_hist = select_gap_coefficients(model_hist, result_hist)
            computation_full = compute_gap_distribution(
                result_full, gap_override=gap, betas_t=beta, topix_night_t=topix,
                vol_adjusted_target=model_full.vol_adjusted_target,
                gap_open_coef=coef_full[0], topix_beta_coef=coef_full[1],
            )
            computation_hist = compute_gap_distribution(
                result_hist, gap_override=gap, betas_t=beta, topix_night_t=topix,
                vol_adjusted_target=model_hist.vol_adjusted_target,
                gap_open_coef=coef_hist[0], topix_beta_coef=coef_hist[1],
            )
            cache_mu = np.load(cache_dir / "matrices" / (f"mu_gap_{date.strftime('%Y%m%d')}.npy" if h == 1 else f"mu_gap_h{h}_{date.strftime('%Y%m%d')}.npy"))
            cache_omega = np.load(cache_dir / "matrices" / (f"omega_gap_{date.strftime('%Y%m%d')}.npy" if h == 1 else f"omega_gap_h{h}_{date.strftime('%Y%m%d')}.npy"))
            call_idx = (0 if date == pd.Timestamp("2026-08-14") else 3) + (0 if h == 1 else 1 if h == 3 else 2)
            captured_call = captured.get("calls", [])[call_idx] if len(captured.get("calls", [])) > call_idx else None
            captured_mu_raw = np.asarray(captured_call.get("mu_raw"), dtype=float) if captured_call else np.array([])
            captured_mu_gap = np.asarray(captured_call.get("mu_gap"), dtype=float) if captured_call else np.array([])
            captured_gap = np.asarray(captured_call.get("gap_override"), dtype=float) if captured_call else np.array([])
            captured_beta = np.asarray(captured_call.get("betas_t"), dtype=float) if captured_call else np.array([])
            row["horizons"][str(h)] = {
                "target": _diff(target_full[: index_full + 1], target_hist[: index_hist + 1]),
                "jp_res_history": _diff(in_full["jp_res_returns_p3"][: index_full + 1], in_hist["jp_res_returns_p3"][: index_hist + 1]),
                "blpx_mu_raw": _diff(computation_full.mu_raw, computation_hist.mu_raw),
                "blpx_omega_raw": _diff(computation_full.omega_raw, computation_hist.omega_raw),
                "gap_mu": _diff(computation_full.mu_gap, computation_hist.mu_gap),
                "gap_omega": _diff(computation_full.omega_gap, computation_hist.omega_gap),
                "cache_vs_full_mu_gap": _diff(cache_mu, computation_full.mu_gap),
                "cache_vs_full_omega_gap": _diff(cache_omega, computation_full.omega_gap),
                "cache_vs_ondemand_mu_gap": _diff(cache_mu, computation_hist.mu_gap),
                "cache_vs_ondemand_omega_gap": _diff(cache_omega, computation_hist.omega_gap),
                "gap_input_snapshot_vs_frame": _diff(gap, frame_gap),
                "beta_input_snapshot_vs_frame": _diff(beta, frame_beta),
                "topix_input_snapshot_vs_frame": abs(topix - frame_topix),
                "frame_topix_oc_return": row_input.get("topix_oc_return"),
                "frame_topix_cc_trade": row_input.get("topix_cc_trade"),
                "coeff_full": list(coef_full),
                "coeff_hist": list(coef_hist),
                "gap_first": float(gap[0]),
                "beta_first": float(beta[0]),
                "topix_night": float(topix),
                "z_u_mean_full": float(np.nanmean(result_full["z_U_t"])),
                "z_u_mean_hist": float(np.nanmean(result_hist["z_U_t"])),
                "mu_raw_first_full": float(computation_full.mu_raw[0]),
                "mu_gap_first_full": float(computation_full.mu_gap[0]),
                "cache_mu_first": float(cache_mu[0]),
                "captured_vs_full_mu_raw": _diff(captured_mu_raw, computation_full.mu_raw) if captured_call else None,
                "captured_vs_full_mu_gap": _diff(captured_mu_gap, computation_full.mu_gap) if captured_call else None,
                "captured_vs_stage_gap": _diff(captured_gap, gap) if captured_call else None,
                "captured_vs_stage_beta": _diff(captured_beta, beta) if captured_call else None,
                "target_full_fingerprint": dataframe_fingerprint(pd.DataFrame(target_full))[:16],
            }
        rows.append(row)
    out = root / "reports/20260920_structural_completion/evidence/r3_stage_decomposition.json"
    out.write_text(json.dumps({"dates": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dates": rows}, indent=2))


if __name__ == "__main__":
    main()
