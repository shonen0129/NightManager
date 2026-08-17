"""Save the df_exec built with the new (shifted) preprocessor to local caches.

This ensures downstream tools (e.g. train_ml_order_overlay.py and
compute_gap_adjusted_distribution.py) use the new-preprocessor df_exec without
re-running preprocess_data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.decision_cache import save_decision_cache
from leadlag.data.market_data_cache import save_df_exec_to_local_cache


def main() -> int:
    df = pd.read_pickle(ROOT / "var" / "results" / "beta_shift_comparison" / "df_exec_new.pkl")
    save_df_exec_to_local_cache(df)
    save_decision_cache(df)
    print(f"Saved {len(df)} rows to df_exec local cache and decision cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
