"""Legacy daily decision result formatter.

Provides ``generate_daily_decision_results()`` used by some unit tests.
The CLI production decision path has moved to ``v2_bridge.run_v2_decision``.
"""

from __future__ import annotations

import logging

import pandas as pd

from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)


import numpy as np


def generate_daily_decision_results(model, df_exec, trade_date, current_weights=None):
    if trade_date == "latest":
        i = len(df_exec) - 1
        trade_date = df_exec.index[i]
    else:
        trade_date = pd.to_datetime(trade_date)
        i = df_exec.index.get_loc(trade_date)
    sig_date = df_exec["sig_date"].values[i]

    pred = model.predict_signals(df_exec)

    raw_pca_sig = pred["raw_pca_signals"].iloc[i].values
    residual_pca_sig = pred["residual_pca_signals"].iloc[i].values
    s_ens = pred["signals"].iloc[i].values

    z0 = model.normalize_signals(raw_pca_sig, model.normalization_method)
    z3 = model.normalize_signals(residual_pca_sig, model.normalization_method)

    w = model.build_weights(s_ens)
    ranks = pd.Series(s_ens).rank(ascending=False).values.astype(int)

    side = []
    for weight in w:
        if weight > 1e-10:
            side.append("LONG")
        elif weight < -1e-10:
            side.append("SHORT")
        else:
            side.append("NEUTRAL")

    sig_records = []
    for j, tk in enumerate(JP_TICKERS):
        rec = {
            "signal_date": sig_date,
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "ticker": tk,
            "production_signal": float(raw_pca_sig[j]),
            "residual_signal": float(residual_pca_sig[j]),
            "production_z": float(z0[j]),
            "residual_z": float(z3[j]),
            "ensemble_signal": float(s_ens[j]),
            "rank": int(ranks[j]),
            "side": side[j],
        }
        if getattr(model, "us_res_enabled", False):
            p4_sig = pred["p4_signals"].iloc[i].values
            z4 = model.normalize_signals(p4_sig, model.normalization_method)
            rec["us_residual_signal"] = float(p4_sig[j])
            rec["us_residual_z"] = float(z4[j])
        sig_records.append(rec)
    latest_signal_df = pd.DataFrame(sig_records)

    gross_exp = float(np.sum(np.abs(w)))
    net_exp = float(np.sum(w))

    weight_records = []
    for j, tk in enumerate(JP_TICKERS):
        weight_records.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "ticker": tk,
                "ensemble_signal": float(s_ens[j]),
                "weight": float(w[j]),
                "side": side[j],
                "gross_exposure": gross_exp,
                "net_exposure": net_exp,
            }
        )
    latest_weights_df = pd.DataFrame(weight_records)

    order_records = []
    for j, tk in enumerate(JP_TICKERS):
        curr_w = float(current_weights.get(tk, 0.0)) if current_weights is not None else 0.0
        target_w = float(w[j])
        delta_w = target_w - curr_w

        note = ""
        if side[j] == "LONG":
            note = "Buy to target weight" if delta_w > 0 else "Reduce long weight"
        elif side[j] == "SHORT":
            note = "Sell to target weight" if delta_w < 0 else "Cover short weight"
        else:
            note = "Close position" if abs(curr_w) > 1e-10 else "No position"

        order_records.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "ticker": tk,
                "current_weight": curr_w,
                "target_weight": target_w,
                "delta_weight": delta_w,
                "side": side[j],
                "note": note,
            }
        )
    latest_orders_df = pd.DataFrame(order_records)

    return {
        "signal_df": latest_signal_df,
        "weights_df": latest_weights_df,
        "orders_df": latest_orders_df,
        "trade_date": trade_date,
        "sig_date": sig_date,
    }
