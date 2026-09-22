"""Inspect the research gap-generator frame versus the production cache frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from research.diagnostics.gap_inputs import attach_topix_trade_returns, load_gap_execution_inputs


def main() -> None:
    local = load_df_exec_from_local_cache()
    loaded = load_gap_execution_inputs()
    research = attach_topix_trade_returns(loaded.df_exec, loaded.raw_data)
    rows = []
    for date in (pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-17")):
        local_row = local.loc[date]
        research_row = research.loc[date]
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "topix_night_local": local_row.get("topix_night_return"),
            "topix_night_research": research_row.get("topix_night_return"),
            "topix_oc_local": local_row.get("topix_oc_return"),
            "topix_oc_research": research_row.get("topix_oc_return"),
            "topix_cc_local": local_row.get("topix_cc_trade"),
            "topix_cc_research": research_row.get("topix_cc_trade"),
            "gap_max_abs": float(np.nanmax(np.abs(local.loc[date, [f"jp_gap_{tk}" for tk in JP_TICKERS]].to_numpy(dtype=float) - research.loc[date, [f"jp_gap_{tk}" for tk in JP_TICKERS]].to_numpy(dtype=float)))),
            "beta_max_abs": float(np.nanmax(np.abs(local.loc[date, [f"jp_beta_{tk}" for tk in JP_TICKERS]].to_numpy(dtype=float) - research.loc[date, [f"jp_beta_{tk}" for tk in JP_TICKERS]].to_numpy(dtype=float)))),
        })
    out = Path("reports/20260920_structural_completion/evidence/r3_source_frame.json")
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
